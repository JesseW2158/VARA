"""Capture overhead marker pixels for the table checks: visibility, printed sizes, check spots. No arm access.

`python scripts/capture_table.py --out DIR` saves one overhead frame (`physical-original.png`, untouched),
an annotated copy and `markers.json` with the observed pixel center and mean edge length of every detected
marker. Physical table coordinates are never inferred; they are tape-measured separately. Flags:

    --frames N                 median marker corners over N reads (default 1); the saved image is the last
    --expect-all               compare the detected ids with every id the camera may see (calibration
                               markers 0-7 and every config/objects.yaml id except the wristband); record
                               expected / missing / unexpected and exit 1 when any is missing (B3's
                               "detected from the camera's height" and B4's "frame at home shows all")
    --expect-hand              with --expect-all, also expect the wristband (id 40)
    --printed-side-mm ID=MM    record ruler-measured printed marker sides (millimeters, verbatim) next to
                               the nominal sides of the make_markers manifest; refused for an id the
                               manifest lacks or a value under half or over twice the nominal (a unit slip)
    --manifest PATH            the scripts/make_markers.py manifest (default runs/markers/markers.json,
                               its default output; when that is absent, the sides make_markers prints)
    --check-point X Y          append the loose marker's observed pixel center with its tape-measured table
                               point [X, Y] (meters) to the check-points list; refused when the marker is
                               not detected, detected more than once, its table point is already in the
                               list or its pixel moved under 0.5 px from a listed one
    --spot-id ID               the loose marker that marks the check spot (default 10; not 0-7)
    --append PATH              the check-points list (default measurements/check-points.json, created with
                               its folder); it keeps exactly {pixel, table} per entry, the format
                               `vision.calibrate --check-points` reads, and the full record (spot id,
                               image, time, mode) goes to the sidecar PATH-stem`.provenance.json`
    --fake                     explicit offline capture through the FakeCamera; every artifact is labeled
                               synthetic and a check-points list must be named synthetic-*.json

markers.json also records `camera_pose`: config/scene.yaml's `Hc` and `c`, and `placeholder`, true while
that file's leading comment or its `camera:` block still calls the values placeholders.

Lens (B5's fallback): when config/lens.npz exists, `vision.calibrate.load_active_lens()` loads it once,
before the camera opens (a corrupt file is refused, exit 2), and every frame goes through
`vision.calibrate.undistort_frame` before marker detection, so the pixels in markers.json and the appended
check points match a homography that `vision.calibrate --live` fitted on undistorted frames.
`physical-original.png` stays the untouched raw frame (the vision.lens fit reads it); the annotated copy
shows the undistorted frame. markers.json and every provenance record carry `lens`: the lens file's
sha256, or null for raw frames (always null with --fake: the FakeCamera is an ideal pinhole, never
undistorted and the file is never read). The CLI prints the same as a `lens:` line. A check point is
refused before the camera opens when the list's provenance records were captured with another lens
(records without `lens` predate the wiring and were raw): after creating or replacing config/lens.npz,
refit the homography and capture the check spots into a new list. A frame whose size differs from the
lens's image size fails the capture (exit 2); move config/lens.npz aside to photograph a checkerboard in
another camera mode.

FAKE / FAKE_CAMERA in the environment cannot silently produce physical evidence: without --fake they are
refused. Every input is checked before the camera opens. Exit codes: 0 ok, 1 a check failed (nothing
detected, an expected marker missing, a check point not appended), 2 bad usage, a refused input or a
capture failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import calibrate, cameras, markers  # noqa: E402

DEFAULT_CHECK_POINTS = ROOT / "measurements" / "check-points.json"
DEFAULT_MANIFEST = ROOT / "runs" / "markers" / "markers.json"  # scripts/make_markers.py's default output
PRINTED_RATIO_RANGE = (0.5, 2.0)  # measured / nominal side outside this is a unit slip, not a print error
SYNTHETIC_PREFIX = "synthetic-"
MAX_TABLE_ABS_M = 2.0  # a table coordinate beyond this is not meters (centimeters typed by mistake)
SAME_PIXEL_PX = 0.5  # a new check spot whose pixel moved less than this is the same spot again


@dataclass(frozen=True)
class CheckPoint:
    """One check spot to record: the tape-measured table point (meters) of the loose marker `spot_id`,
    appended to the list at `path` (None: DEFAULT_CHECK_POINTS, looked up when the capture runs)."""

    table: tuple[float, float]
    spot_id: int = 10
    path: Path | None = None


def expected_ids(
    *, include_hand: bool = False, table: Mapping[int, Mapping[str, Any]] | None = None
) -> list[int]:
    """Every id the overhead camera is expected to see: `vision.markers.marker_table()` (calibration
    markers 0-7 plus config/objects.yaml), without the wristband unless `include_hand`."""
    entries = table if table is not None else markers.marker_table()
    return sorted(int(i) for i, entry in entries.items() if include_hand or entry.get("kind") != "hand")


def parse_printed_sides(values: Sequence[str]) -> dict[int, float]:
    """`["10=30.5", "0=60"]` -> {10: 30.5, 0: 60.0}: ruler-measured printed sides in millimeters.

    Each value is ID=MM with a positive finite MM, each id at most once (ValueError otherwise). Which ids
    exist and their nominal sides come from `nominal_sides`.
    """
    sides: dict[int, float] = {}
    for text in values:
        key, sep, number = str(text).partition("=")
        if not sep:
            raise ValueError(f"printed side {text!r} must look like ID=MM (millimeters)")
        try:
            marker_id, mm = int(key.strip()), float(number.strip())
        except ValueError as exc:
            raise ValueError(f"printed side {text!r} must look like ID=MM (millimeters)") from exc
        if not math.isfinite(mm) or mm <= 0.0:
            raise ValueError(f"printed side of marker {marker_id} must be positive millimeters, got {mm!r}")
        if marker_id in sides:
            raise ValueError(f"marker id {marker_id} given twice")
        sides[marker_id] = mm
    return sides


def nominal_sides(manifest: Path | str | None = None) -> tuple[dict[int, float], str]:
    """The nominal printed side (millimeters) per marker id, and where it came from.

    Read from the scripts/make_markers.py manifest (`markers[].id`, `markers[].side_cm`): `manifest`, or
    DEFAULT_MANIFEST when that file exists. Without either, the sides make_markers prints by kind, for
    the ids of `vision.markers.marker_table()` (calibration ids 0-7 and config/objects.yaml). A given
    manifest that is missing or malformed raises ValueError.
    """
    file = Path(manifest) if manifest is not None else DEFAULT_MANIFEST
    if manifest is None and not file.exists():
        table = markers.marker_table()
        sides = {int(i): round(float(entry["side"]) * 1000.0, 4) for i, entry in table.items()}
        return sides, f"vision.markers sides by kind (no make_markers manifest at {_evidence_path(file)})"
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"make_markers manifest {file} does not exist") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"make_markers manifest {file} is not readable JSON: {exc}") from exc
    entries = data.get("markers") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{file} is not a make_markers manifest (no markers list)")
    sides = {}
    for index, entry in enumerate(entries):
        marker_id = entry.get("id") if isinstance(entry, dict) else None
        side_cm = entry.get("side_cm") if isinstance(entry, dict) else None
        if (
            isinstance(marker_id, bool)
            or not isinstance(marker_id, int)
            or isinstance(side_cm, bool)
            or not isinstance(side_cm, (int, float))
            or not math.isfinite(side_cm)
            or side_cm <= 0
        ):
            raise ValueError(f"{file}: marker entry {index} needs an integer id and a positive side_cm")
        if marker_id in sides:
            raise ValueError(f"{file}: marker id {marker_id} is listed twice")
        sides[marker_id] = round(float(side_cm) * 10.0, 4)
    return sides, _evidence_path(file)


def printed_side_entries(
    measured: Mapping[int, float], nominal: Mapping[int, float], source: str
) -> list[dict[str, Any]]:
    """`[{"id", "measured_mm", "nominal_mm"}]` sorted by id, the measurements recorded verbatim.

    Refuses an id without a nominal side in `nominal` (described by `source`) and a measurement outside
    PRINTED_RATIO_RANGE times its nominal side, which is a unit slip (centimeters typed for millimeters).
    """
    low, high = PRINTED_RATIO_RANGE
    entries: list[dict[str, Any]] = []
    for marker_id, mm in sorted(measured.items()):
        if int(marker_id) not in nominal:
            raise ValueError(f"marker id {marker_id} has no nominal side in {source}")
        value, expected = float(mm), float(nominal[int(marker_id)])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"printed side of marker {marker_id} must be positive millimeters, got {mm!r}")
        if not low <= value / expected <= high:
            raise ValueError(
                f"printed side {value:g} mm of marker {marker_id} is far from its nominal {expected:g} mm: "
                "give millimeters (did you type centimeters?)"
            )
        entries.append({"id": int(marker_id), "measured_mm": value, "nominal_mm": expected})
    return entries


# -- the check-points list -----------------------------------------------------------------------------


def provenance_path(path: Path | str) -> Path:
    """`measurements/check-points.json` -> `measurements/check-points.provenance.json`."""
    file = Path(path)
    return file.with_name(f"{file.stem}.provenance.json")


def _pair(value: Any, what: str, index: int, where: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise ValueError(f"{where}: check point {index} {what} must be two finite numbers")
    pair = [float(v) for v in value]
    if not all(math.isfinite(v) for v in pair):
        raise ValueError(f"{where}: check point {index} {what} must be two finite numbers")
    return pair


def load_check_point_list(path: Path | str) -> list[dict[str, list[float]]]:
    """The check-points list at `path` ([] when the file does not exist yet), every entry validated to
    exactly `{"pixel": [u, v], "table": [x, y]}` of finite numbers (what vision.calibrate accepts)."""
    file = Path(path)
    if not file.exists():
        return []
    try:
        points = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file} is not valid JSON: {exc}") from exc
    if not isinstance(points, list):
        raise ValueError(f"{file} must hold a JSON list of check points")
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {"pixel", "table"}:
            raise ValueError(f"{file}: check point {index} needs exactly pixel: [u, v] and table: [x, y]")
        _pair(point["pixel"], "pixel", index, str(file))
        _pair(point["table"], "table", index, str(file))
    return points


def load_provenance(path: Path | str) -> list[dict[str, Any]]:
    """The provenance records of the check-points list at `path` ([] when its sidecar is absent).

    A sidecar whose list no longer exists is refused (ValueError), so the records of an old, deleted list
    never mix with a new session's; so is a sidecar that is not a JSON list of objects.
    """
    file = Path(path)
    sidecar = provenance_path(file)
    if not sidecar.exists():
        return []
    if not file.exists():
        raise ValueError(
            f"{sidecar} exists without {file.name}: move that old provenance away before starting a new list"
        )
    try:
        records = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar} is not valid JSON: {exc}") from exc
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{sidecar} must hold a JSON list of provenance records")
    return records


def _checked_table(table: Sequence[float]) -> list[float]:
    try:
        x, y = float(table[0]), float(table[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"table point must be [x, y] in meters, got {table!r}") from exc
    if len(table) != 2 or not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"table point must be two finite meters, got {table!r}")
    if abs(x) > MAX_TABLE_ABS_M or abs(y) > MAX_TABLE_ABS_M:
        raise ValueError(f"table point {[x, y]} is not in meters (did you type centimeters?)")
    return [x, y]


def _refuse_duplicates(
    points: Sequence[Mapping[str, Sequence[float]]],
    table: Sequence[float],
    pixel: Sequence[float] | None,
    where: str,
) -> None:
    for index, point in enumerate(points):
        if all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(point["table"], table, strict=True)):
            raise ValueError(
                f"table point {list(table)} is already entry {index} of {where}; measure a new spot"
            )
        if pixel is not None and math.dist(point["pixel"], pixel) < SAME_PIXEL_PX:
            raise ValueError(
                f"pixel {list(pixel)} is within {SAME_PIXEL_PX} px of entry {index} of {where}: "
                "the marker did not move"
            )


def _write_json_atomic(file: Path, data: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=file.parent,
            prefix=f".{file.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _lens_text(lens: str | None) -> str:
    return "null (raw frames)" if lens is None else lens


def _refuse_lens_mix(records: Sequence[Mapping[str, Any]], lens: str | None, file: Path) -> None:
    """Raw and undistorted pixels never share a list: ValueError when a provenance record's `lens` differs
    from this capture's `lens` (a record without the key predates the lens wiring, so its pixel is raw)."""
    for position, record in enumerate(records):
        recorded = record.get("lens")
        if recorded != lens:
            index = record.get("index", position)
            raise ValueError(
                f"entry {index} of {file.name} was captured with lens {_lens_text(recorded)} and this "
                f"capture uses lens {_lens_text(lens)}: raw and undistorted pixels do not mix. After "
                "creating or replacing config/lens.npz, refit the homography and start a new list (move "
                f"{file.name} and its provenance sidecar away)"
            )


def append_check_point(
    path: Path | str,
    pixel: Sequence[float],
    table: Sequence[float],
    *,
    spot_id: int,
    image: str,
    mode: str,
    t: str,
    lens: str | None = None,
) -> dict[str, Any]:
    """Append `{"pixel": [u, v], "table": [x, y]}` to the list at `path` (created with its parent when
    absent) and the full record `{index, pixel, table, spot_id, image, t, mode, lens}` to its provenance
    sidecar, both written atomically. `lens` is the sha256 of the lens the pixel was undistorted with
    (`vision.calibrate.lens_hash`), None for a raw frame; the list itself keeps exactly pixel and table,
    the only keys `vision.calibrate.load_check_points` accepts. Refuses a table point already recorded, a
    pixel that did not move or a `lens` other than the earlier records'. Returns `{"index", "count",
    "file", "provenance"}`."""
    file = Path(path)
    points = load_check_point_list(file)
    u, v = float(pixel[0]), float(pixel[1])
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"pixel must be two finite numbers, got {pixel!r}")
    if lens is not None and not isinstance(lens, str):
        raise ValueError(f"lens must be a sha256 string or None, got {lens!r}")
    xy = _checked_table(table)
    _refuse_duplicates(points, xy, [u, v], str(file))
    sidecar = provenance_path(file)
    provenance = load_provenance(file)
    _refuse_lens_mix(provenance, lens, file)
    points.append({"pixel": [u, v], "table": xy})
    index = len(points) - 1
    provenance.append(
        {
            "index": index,
            "pixel": [u, v],
            "table": xy,
            "spot_id": int(spot_id),
            "image": str(image),
            "t": str(t),
            "mode": str(mode),
            "lens": lens,
        }
    )
    _write_json_atomic(file, points)
    _write_json_atomic(sidecar, provenance)
    return {"index": index, "count": len(points), "file": str(file), "provenance": str(sidecar)}


def _check_point_plan(check_point: CheckPoint, fake: bool, lens: str | None = None) -> dict[str, Any]:
    """Validate a requested check point before the camera opens: id, table pair, file label, duplicates
    and the lens of the earlier provenance records (`lens`: this capture's lens sha256, None for raw)."""
    spot_id = check_point.spot_id
    if isinstance(spot_id, bool) or not isinstance(spot_id, int) or not 0 <= spot_id < 50:
        raise ValueError(f"spot id must be a DICT_4X4_50 id (0-49), got {spot_id!r}")
    if spot_id in markers.CALIB_IDS:
        raise ValueError(
            f"spot id {spot_id} is a fitting marker; mark check spots with a loose object marker"
        )
    table = _checked_table(check_point.table)
    file = Path(check_point.path) if check_point.path is not None else DEFAULT_CHECK_POINTS
    if file.suffix.lower() != ".json":
        raise ValueError(f"the check-points list must be a .json file, got {file}")
    labeled = file.name.startswith(SYNTHETIC_PREFIX)
    if fake and not labeled:
        raise ValueError(
            f"a --fake check point must go to a list named {SYNTHETIC_PREFIX}*.json, not {file.name}: "
            "synthetic pixels are not table measurements"
        )
    if not fake and labeled:
        raise ValueError(
            f"{file.name} is labeled synthetic; physical check points go to a list without that prefix"
        )
    existing = load_check_point_list(file)
    _refuse_lens_mix(load_provenance(file), lens, file)
    _refuse_duplicates(existing, table, None, str(file))
    return {"spot_id": spot_id, "table": table, "file": file, "count": len(existing)}


# -- capture ---------------------------------------------------------------------------------------------


def _id_counts(frame: np.ndarray, det: Any) -> dict[int, int]:
    """How many times each id appears in the raw detector output (`markers.detect` keeps one per id)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _corners, ids, _rejected = det.detectMarkers(gray)
    counts: dict[int, int] = {}
    if ids is not None:
        for marker_id in ids.ravel().tolist():
            counts[int(marker_id)] = counts.get(int(marker_id), 0) + 1
    return counts


def _mean_edge_px(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.mean([np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]))


def camera_pose_placeholder(scene_text: str) -> bool:
    """True while config/scene.yaml's leading comment or its top-level `camera:` block (the key line,
    its indented values and their comments) still says "placeholder". Other blocks' notes do not count."""
    lines = scene_text.splitlines()
    header: list[str] = []
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            break
        header.append(line)
    block: list[str] = []
    inside = False
    for line in lines:
        if line[:1] and not line[:1].isspace() and not line.startswith("#"):
            inside = line.startswith("camera:")
        if inside:
            block.append(line)
    return any("placeholder" in line.lower() for line in (*header, *block))


def _camera_pose_block() -> dict[str, Any]:
    """`Hc`, `c` from config/scene.yaml and whether the file still calls them placeholders."""
    cfg = calibrate.load_scene_cfg()
    c, hc = calibrate.camera_pose(cfg)
    text = calibrate.SCENE_PATH.read_text(encoding="utf-8")
    return {
        "Hc": hc,
        "c": [c[0], c[1]],
        "source": _evidence_path(calibrate.SCENE_PATH),
        "placeholder": camera_pose_placeholder(text),
    }


def _evidence_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def capture(
    out: Path,
    *,
    fake: bool = False,
    frames: int = 1,
    expect: Sequence[int] | None = None,
    printed_side_mm: Mapping[int, float] | None = None,
    manifest: Path | str | None = None,
    check_point: CheckPoint | None = None,
) -> dict[str, Any]:
    """Save one frame and the detected marker pixels in a new directory; always release the camera.

    Physical coordinates must be measured separately. FAKE / FAKE_CAMERA cannot silently produce
    physical evidence; synthetic capture requires ``fake=True`` and is labeled in every artifact.
    `frames` > 1 takes the median corners per id over that many reads. `expect` adds `expected`,
    `missing` and `unexpected` ids to the report; `printed_side_mm` records ruler measurements next to
    the nominal sides of `manifest` (see `nominal_sides`; the source lands in `nominal_side_source`);
    `check_point` appends the spot marker's pixel with its table point (see `append_check_point`; a
    refusal after the capture is recorded under `check_point.reason`, never raised). Every input is
    validated before the camera opens (ValueError, nothing written), config/lens.npz included: a physical
    capture loads it once with `vision.calibrate.load_active_lens` (corrupt: ValueError) and undistorts
    every frame before detection; the report's `lens` is its sha256, or None for raw frames and always
    for ``fake=True`` (the file is then never read). A frame the lens cannot take is a RuntimeError.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory")
    if not fake and any(os.environ.get(flag) == "1" for flag in ("FAKE", "FAKE_CAMERA")):
        raise ValueError("unset FAKE and FAKE_CAMERA for physical capture, or use --fake for offline checks")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError(f"frames must be a positive integer, got {frames!r}")
    expected = sorted({int(i) for i in expect}) if expect is not None else None
    if manifest is not None and printed_side_mm is None:
        raise ValueError("a make_markers manifest only supplies nominal sides for printed_side_mm")
    sides: list[dict[str, Any]] | None = None
    side_source: str | None = None
    if printed_side_mm is not None:
        nominal, side_source = nominal_sides(manifest)
        sides = printed_side_entries(printed_side_mm, nominal, side_source)
    lens = None if fake else calibrate.load_active_lens()  # once, before the camera opens
    lens_sha = calibrate.lens_hash(lens)
    plan = _check_point_plan(check_point, fake, lens_sha) if check_point is not None else None
    camera_pose = _camera_pose_block()
    camera = cameras.FakeCamera() if fake else cameras.get_camera("overhead")
    try:
        if not fake and getattr(camera, "is_fake", True) is not False:
            raise ValueError("physical capture received a synthetic or unidentified camera")
        det = markers.detector()
        seen: dict[int, list[np.ndarray]] = {}
        counts: dict[int, int] = {}
        raw: np.ndarray | None = None
        frame: np.ndarray | None = None
        for _ in range(frames):
            raw = camera.read()
            if raw is None or raw.size == 0:
                raise RuntimeError("overhead camera returned no frame")
            frame = _undistorted(raw, lens)
            for marker_id, corners in markers.detect(frame):
                seen.setdefault(int(marker_id), []).append(
                    np.asarray(corners, dtype=np.float64).reshape(4, 2)
                )
            for marker_id, count in _id_counts(frame, det).items():
                counts[marker_id] = max(counts.get(marker_id, 0), count)
        assert raw is not None and frame is not None
        detections = [
            (marker_id, np.median(np.stack(quads), axis=0).astype(np.float32))
            for marker_id, quads in sorted(seen.items())
        ]
        observations = [
            {
                "id": int(marker_id),
                "pixel": corners.mean(axis=0).astype(float).tolist(),
                "edge_px": _mean_edge_px(corners),
                "frames_seen": len(seen[marker_id]),
            }
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
        detected_ids = [entry["id"] for entry in observations]
        report: dict[str, Any] = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "camera": "overhead",
            "camera_index": getattr(camera, "index", None),
            "camera_pose": camera_pose,
            "dictionary": "DICT_4X4_50",
            "image_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "frames": int(frames),
            "pixel_origin": "top-left; u right, v down",
            "center_method": "mean of four detected marker corners"
            + (" on frames undistorted through config/lens.npz" if lens is not None else "")
            + (f", median over {frames} frames" if frames > 1 else ""),
            "lens": lens_sha,
            "markers": observations,
            "duplicate_ids": sorted(marker_id for marker_id, count in counts.items() if count > 1),
            "original_image": f"{mode}-original.png",
            "annotated_image": f"{mode}-annotated.png",
            "note": "Observed pixels only. Tape-measure table coordinates separately; no calibration saved.",
        }
        if fake and hasattr(camera, "hidden_calibration_ids"):
            report["synthetic_hidden_calibration_ids"] = sorted(camera.hidden_calibration_ids())
        if expected is not None:
            report["expected"] = expected
            report["missing"] = [marker_id for marker_id in expected if marker_id not in detected_ids]
            report["unexpected"] = [marker_id for marker_id in detected_ids if marker_id not in expected]
        if sides is not None:
            report["printed_side_mm"] = sides
            report["nominal_side_source"] = side_source
        if plan is not None:
            report["check_point"] = {
                "spot_id": plan["spot_id"],
                "table": plan["table"],
                "file": str(plan["file"]),
                "provenance": str(provenance_path(plan["file"])),
                "appended": False,
                "index": None,
                "count": plan["count"],
                "reason": None,
            }
        # Encode before creating the directory so an image failure cannot look like completed evidence.
        images = []
        for key, image in (("original_image", raw), ("annotated_image", annotated)):
            ok, data = cv2.imencode(".png", image)
            if not ok:
                raise OSError("could not encode capture image")
            images.append((report[key], data.tobytes()))
        out.mkdir(parents=True, exist_ok=False)
        for name, data in images:
            with (out / name).open("xb") as stream:
                stream.write(data)
        if plan is not None:
            _append_from_capture(report, plan, observations, counts, out)
        with (out / "markers.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        camera.release()


def _undistorted(frame: np.ndarray, lens: Mapping[str, Any] | None) -> np.ndarray:
    """`vision.calibrate.undistort_frame`, its size refusal turned into a capture failure (RuntimeError)."""
    try:
        return calibrate.undistort_frame(frame, lens)
    except ValueError as exc:
        raise RuntimeError(
            f"{exc}; to photograph a checkerboard in this camera mode, move config/lens.npz aside first"
        ) from exc


def lens_line(report: Mapping[str, Any]) -> str:
    """The `lens:` line the CLI prints for a capture report (its `lens` and `mode`)."""
    if report.get("mode") == "synthetic":
        return "lens: null (synthetic FakeCamera: an ideal pinhole, never undistorted)"
    if report.get("lens") is None:
        return "lens: null (no config/lens.npz: markers detected on raw frames)"
    return (
        f"lens: {report['lens']} (config/lens.npz: every frame undistorted before detection; "
        "the original image stays raw)"
    )


def _append_from_capture(
    report: dict[str, Any],
    plan: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    counts: Mapping[int, int],
    out: Path,
) -> None:
    """Append the spot marker's pixel to the list, or record why not (`check_point.reason`)."""
    block = report["check_point"]
    spot_id = int(plan["spot_id"])
    observation = next((entry for entry in observations if entry["id"] == spot_id), None)
    count = counts.get(spot_id, 0)
    if observation is None or count == 0:
        block["reason"] = f"marker {spot_id} not detected (seen: {[e['id'] for e in observations]})"
        return
    if count > 1:
        block["reason"] = f"marker {spot_id} detected {count} times; keep other copies of it out of view"
        return
    try:
        result = append_check_point(
            plan["file"],
            observation["pixel"],
            plan["table"],
            spot_id=spot_id,
            image=_evidence_path(out / report["original_image"]),
            mode=report["mode"],
            t=report["captured_at"],
            lens=report["lens"],
        )
    except ValueError as exc:
        block["reason"] = str(exc)
        return
    block.update(
        {"appended": True, "index": result["index"], "count": result["count"], "pixel": observation["pixel"]}
    )


# -- CLI -------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; see the module docstring for the flags and exit codes."""
    parser = argparse.ArgumentParser(
        description="Capture an overhead frame and the detected marker pixels for the table checks (B3-B5)."
    )
    parser.add_argument("--out", type=Path, required=True, help="new directory for images and markers.json")
    parser.add_argument("--fake", action="store_true", help="explicit offline capture; not physical evidence")
    parser.add_argument(
        "--frames", type=int, default=1, metavar="N", help="median marker corners over N reads (default 1)"
    )
    parser.add_argument(
        "--expect-all",
        action="store_true",
        help="exit 1 unless every calibration and objects.yaml marker is seen",
    )
    parser.add_argument(
        "--expect-hand", action="store_true", help="with --expect-all, also expect the wristband"
    )
    parser.add_argument(
        "--printed-side-mm", nargs="+", metavar="ID=MM", help="ruler-measured printed marker sides to record"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help="make_markers manifest with the nominal sides (default runs/markers/markers.json when present)",
    )
    parser.add_argument(
        "--check-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="append the spot marker's pixel with this tape-measured table point (meters)",
    )
    parser.add_argument(
        "--spot-id",
        type=int,
        default=None,
        metavar="ID",
        help="the loose marker on the check spot (default 10)",
    )
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        metavar="PATH",
        help="check-points list to append to (default measurements/check-points.json)",
    )
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.expect_hand and not args.expect_all:
        parser.error("--expect-hand requires --expect-all")
    if args.manifest is not None and not args.printed_side_mm:
        parser.error("--manifest requires --printed-side-mm")
    if args.check_point is None and (args.spot_id is not None or args.append is not None):
        parser.error("--spot-id and --append require --check-point X Y")
    try:
        printed = parse_printed_sides(args.printed_side_mm) if args.printed_side_mm else None
    except ValueError as exc:
        parser.error(str(exc))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        expect = expected_ids(include_hand=args.expect_hand) if args.expect_all else None
        check_point = None
        if args.check_point is not None:
            check_point = CheckPoint(
                (args.check_point[0], args.check_point[1]),
                args.spot_id if args.spot_id is not None else 10,
                args.append,
            )
        report = capture(
            args.out,
            fake=args.fake,
            frames=args.frames,
            expect=expect,
            printed_side_mm=printed,
            manifest=args.manifest,
            check_point=check_point,
        )
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 2
    status = 0
    print(f"Saved {report['mode']} frame to {args.out}; {len(report['markers'])} marker(s) detected.")
    print(lens_line(report))
    for entry in report["markers"]:
        print(f"ID {entry['id']}: pixel {entry['pixel']} edge {entry['edge_px']:.1f} px")
    if report["duplicate_ids"]:
        print(f"Warning: ids seen more than once in the frame: {report['duplicate_ids']}")
    if report.get("synthetic_hidden_calibration_ids"):
        print(
            f"Synthetic scene: calibration marker(s) {report['synthetic_hidden_calibration_ids']} lie under "
            "raised items, where the overhead camera cannot see them."
        )
    if expect is not None:
        if report["missing"]:
            print(f"Missing expected markers: {report['missing']} (expected {report['expected']})")
            status = 1
        else:
            print(f"All {len(report['expected'])} expected markers detected.")
        if report["unexpected"]:
            print(f"Unexpected ids in view: {report['unexpected']}")
    for side in report.get("printed_side_mm") or []:
        measured, nominal = side["measured_mm"], side["nominal_mm"]
        print(f"ID {side['id']}: printed side {measured:g} mm recorded (nominal {nominal:g} mm)")
    if report.get("printed_side_mm"):
        print(f"Nominal sides from {report['nominal_side_source']}.")
    block = report.get("check_point")
    if block is not None:
        if block["appended"]:
            needed = calibrate.MIN_CHECK_POINTS
            count = block["count"]
            progress = f"{count} of {needed} recorded" if count <= needed else f"{count} recorded"
            print(
                f"Check point {block['index'] + 1} appended to {block['file']}: pixel {block['pixel']} "
                f"table {block['table']}; {progress}"
            )
            if count >= needed:
                try:
                    calibrate.load_check_points(block["file"])
                except ValueError as exc:
                    print(f"check points not yet valid: {exc}")
                else:
                    print("check points valid")
        else:
            print(f"Check point NOT appended: {block['reason']}", file=sys.stderr)
            status = 1
    if not report["markers"]:
        print("No marker centers available; adjust the view and capture again in a new directory.")
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
