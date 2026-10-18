"""Lens calibration and undistortion (vision/lens.py) on synthetic checkerboard photos.

No camera and no file outside tmp_path. The views render once per module through `synthesize_views`
(a known pinhole plus k1 = -0.10, rendered by exact inverse mapping, not drawChessboardCorners).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision import lens, markers

PATTERN = lens.DEFAULT_PATTERN
CORNERS = PATTERN[0] * PATTERN[1]


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    return lens.synthesize_views(tmp_path_factory.mktemp("boards"))


@pytest.fixture(scope="module")
def fitted(views):
    return lens.calibrate_lens(views["images"])


def _read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)


def _residual_to_board(corners) -> float:
    """RMS distance (px) of detected corners from the best homography of the flat board: 0 for a pinhole."""
    objp = lens.board_points(PATTERN)[:, :2].astype(np.float64)
    pts = corners.reshape(-1, 2).astype(np.float64)
    h, _ = cv2.findHomography(objp, pts, 0)
    projected = cv2.perspectiveTransform(objp.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - pts) ** 2, axis=1))))


def _good_lens() -> dict:
    return {
        "image_size": [640, 480],
        "camera_matrix": np.array([[450.0, 0, 320.0], [0, 450.0, 240.0], [0, 0, 1.0]]),
        "dist_coeffs": np.zeros(5),
        "rms": 0.1,
    }


# -- synthetic views and the fit -----------------------------------------------------------------------


def test_synthetic_views_are_labeled_and_match_their_projection(views):
    assert len(views["images"]) == lens.SYNTHETIC_VIEWS
    objp = lens.board_points(PATTERN)
    for path, (rvec, tvec) in zip(views["images"], views["poses"], strict=True):
        assert path.name.startswith("synthetic-board-") and path.suffix == ".png"
        image = _read(path)
        assert image.shape == (480, 640) and image.dtype == np.uint8
        corners = lens.find_corners(image, PATTERN)
        assert corners is not None and corners.shape == (CORNERS, 1, 2)
        truth, _ = cv2.projectPoints(objp, rvec, tvec, views["camera_matrix"], views["dist_coeffs"])
        found = corners.reshape(-1, 2)
        forward = np.linalg.norm(found - truth.reshape(-1, 2), axis=1)
        reverse = np.linalg.norm(found[::-1] - truth.reshape(-1, 2), axis=1)  # the 180-degree ambiguity
        errors = forward if forward.mean() <= reverse.mean() else reverse
        # 0.5 px even on the smallest (about 14 px) squares: a fixed 11 px cornerSubPix half-window
        # reached the next corner there and was 2.7 px off.
        assert errors.mean() < 0.25 and errors.max() < 0.5


def test_calibrate_lens_recovers_the_synthetic_camera(views, fitted):
    assert fitted["image_size"] == [640, 480]
    assert abs(fitted["dist_coeffs"][0] - lens.SYNTHETIC_K1) < 0.01
    assert fitted["dist_coeffs"].shape == (5,) and fitted["dist_coeffs"][4] == 0.0
    assert fitted["rms"] < 0.5 and max(fitted["per_image_rms"]) < 0.8
    k = fitted["camera_matrix"]
    assert abs(k[0, 0] - 450.0) < 4.5 and abs(k[1, 1] - 450.0) < 4.5
    assert abs(k[0, 2] - 320.0) < 4.0 and abs(k[1, 2] - 240.0) < 4.0
    assert fitted["synthetic"] is True
    assert fitted["images"] == [p.name for p in views["images"]] and fitted["skipped"] == []
    assert len(fitted["images_sha256"]) == 64
    assert fitted["pattern"] == [9, 6] and fitted["square_m"] == 0.025 and fitted["created_at"]


def test_fewer_than_15_usable_images_is_refused(views, tmp_path):
    paths = views["images"][:14]
    with pytest.raises(ValueError, match="need 15"):
        lens.calibrate_lens(paths)
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((480, 640), 200, np.uint8))
    with pytest.raises(ValueError, match=r"only 14 usable of 15 images.*blank\.png \(no full 9x6 board\)"):
        lens.calibrate_lens([*paths, blank])


def test_repeated_photos_do_not_count_twice(views, tmp_path):
    copy = tmp_path / "again.png"
    copy.write_bytes(views["images"][0].read_bytes())
    with pytest.raises(ValueError, match=r"only 14 usable of 15 images.*again\.png \(same bytes as "):
        lens.calibrate_lens([*views["images"][:14], copy])
    result = lens.calibrate_lens([*views["images"][:15], copy])
    assert len(result["images"]) == 15 and len(result["skipped"]) == 1
    (skipped,) = result["skipped"]
    assert "again.png (same bytes as " in skipped and skipped.endswith("synthetic-board-01.png)")


def test_capture_table_directories_are_read_without_their_annotated_copies(views, tmp_path):
    folder = tmp_path / "lens-photos"
    for index, path in enumerate(views["images"], start=1):
        spot = folder / f"board-{index:02d}"
        spot.mkdir(parents=True)
        (spot / lens.CAPTURE_IMAGE).write_bytes(path.read_bytes())
        (spot / "physical-annotated.png").write_bytes(views["images"][0].read_bytes())
        (spot / "markers.json").write_text("{}", encoding="utf-8")
    (folder / "extra-annotated.png").write_bytes(views["images"][1].read_bytes())
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    paths = lens.list_images(folder)
    assert len(paths) == lens.SYNTHETIC_VIEWS and all(p.name == lens.CAPTURE_IMAGE for p in paths)
    result = lens.calibrate_lens(paths)
    assert result["images"][0] == "board-01/physical-original.png" and result["skipped"] == []
    assert len(set(result["images"])) == lens.SYNTHETIC_VIEWS
    single = lens.list_images(folder / "board-02")  # one capture directory: its original, not the copy
    assert single == [folder / "board-02" / lens.CAPTURE_IMAGE]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .* images and no"):
        lens.list_images(empty)


def test_mixed_sizes_bad_files_and_empty_lists_are_refused(views, tmp_path):
    small = tmp_path / "small.png"
    cv2.imwrite(str(small), np.full((240, 320), 200, np.uint8))
    with pytest.raises(ValueError, match="one camera mode"):
        lens.calibrate_lens([*views["images"], small])
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not an image"):
        lens.calibrate_lens([junk])
    with pytest.raises(ValueError, match="no board images"):
        lens.calibrate_lens([])
    with pytest.raises(FileNotFoundError):
        lens.calibrate_lens([tmp_path / "missing.png"])


# -- save and load --------------------------------------------------------------------------------------


def test_save_and_load_round_trip(fitted, tmp_path):
    path = tmp_path / "config" / "lens.npz"
    assert lens.load_lens(path) is None
    assert lens.save_lens(fitted, path) == path
    assert sorted(p.name for p in path.parent.iterdir()) == ["lens.npz"]  # no temporary file left behind
    loaded = lens.load_lens(path)
    assert np.array_equal(loaded["camera_matrix"], fitted["camera_matrix"])
    assert np.array_equal(loaded["dist_coeffs"], fitted["dist_coeffs"])
    assert loaded["image_size"] == [640, 480] and loaded["rms"] == fitted["rms"]
    assert loaded["per_image_rms"] == pytest.approx(fitted["per_image_rms"])
    assert loaded["images"] == fitted["images"] and loaded["images_sha256"] == fitted["images_sha256"]
    assert loaded["skipped"] == [] and loaded["pattern"] == [9, 6] and loaded["square_m"] == 0.025
    assert loaded["created_at"] == fitted["created_at"] and loaded["synthetic"] is True
    assert loaded["path"] == str(path)


def test_corrupt_or_incomplete_lens_files_raise(tmp_path):
    junk = tmp_path / "lens.npz"
    junk.write_bytes(b"junk")
    with pytest.raises(ValueError, match="not a lens file"):
        lens.load_lens(junk)
    partial = tmp_path / "partial.npz"
    np.savez(partial, camera_matrix=np.eye(3))
    with pytest.raises(ValueError, match="missing"):
        lens.load_lens(partial)
    bad = tmp_path / "bad.npz"
    np.savez(bad, image_size=[640, 480], camera_matrix=np.eye(3), dist_coeffs=np.zeros(3), rms=0.1)
    with pytest.raises(ValueError, match="dist_coeffs"):
        lens.load_lens(bad)


def test_save_lens_rejects_bad_results_and_writes_nothing(tmp_path):
    path = tmp_path / "lens.npz"
    cases = [
        ({"camera_matrix": np.zeros((3, 3))}, "focal"),
        ({"camera_matrix": np.eye(2)}, "3x3"),
        ({"dist_coeffs": np.zeros(3)}, "dist_coeffs"),
        ({"dist_coeffs": np.array([np.nan, 0, 0, 0, 0])}, "dist_coeffs"),
        ({"image_size": [0, 480]}, "positive"),
        ({"rms": float("inf")}, "rms"),
    ]
    for override, message in cases:
        with pytest.raises(ValueError, match=message):
            lens.save_lens({**_good_lens(), **override}, path)
    with pytest.raises(ValueError, match="missing"):
        lens.save_lens({"rms": 0.1}, path)
    assert not path.exists() and list(tmp_path.iterdir()) == []


# -- undistortion ---------------------------------------------------------------------------------------


def _checkerboard(pattern, px_per_square=32):
    """A numpy checkerboard texture: (cols+1) x (rows+1) squares inside a one-square white margin, so the
    inner corners sit at whole squares from the texture's third square (as in `lens.board_points`)."""
    cols, rows = pattern
    texture = np.full(((rows + 3) * px_per_square, (cols + 3) * px_per_square), 255, np.uint8)
    for b in range(-1, rows):
        for a in range(-1, cols):
            if (a + b) % 2 == 0:
                top, left = (b + 2) * px_per_square, (a + 2) * px_per_square
                texture[top : top + px_per_square, left : left + px_per_square] = 0
    return texture


def test_undistort_straightens_a_frame_filling_board(views, fitted):
    square, (cols, rows) = lens.DEFAULT_SQUARE_M, PATTERN
    rvec = np.array([0.04, -0.03, 0.02])
    rotation, _ = cv2.Rodrigues(rvec)
    center = np.array([(cols - 1) * square / 2.0, (rows - 1) * square / 2.0, 0.0])
    tvec = np.array([0.0, 0.0, 0.23]) - rotation @ center  # 0.23 m away: the board fills the frame
    origin, extent = (-2.0 * square, -2.0 * square), ((cols + 3) * square, (rows + 3) * square)
    k_true, d_true = views["camera_matrix"], views["dist_coeffs"]
    image = lens.render_plane(_checkerboard(PATTERN), origin, extent, k_true, d_true, rvec, tvec)
    raw = _residual_to_board(lens.find_corners(image, PATTERN))
    flat = _residual_to_board(lens.find_corners(lens.undistort(image, fitted), PATTERN))
    assert raw > 0.8  # k1 = -0.10 bends the board's rows by about a pixel: no homography fits them
    assert flat < 0.2  # undistorted, the board is flat again (corner noise only)
    for path in views["images"]:  # and no view of the fit comes out worse than corner noise
        undistorted = lens.undistort(_read(path), fitted)
        assert _residual_to_board(lens.find_corners(undistorted, PATTERN)) < 0.3


def test_undistorted_marker_lands_on_its_undistorted_pixel(views, fitted):
    side, k = 0.04, 24
    patch = cv2.copyMakeBorder(markers.generate(10, 6 * k), k, k, k, k, cv2.BORDER_CONSTANT, value=255)
    origin, extent = (-side / 6.0, -side / 6.0), (side * 8.0 / 6.0, side * 8.0 / 6.0)
    center = np.array([[side / 2.0, side / 2.0, 0.0]])
    rvec = np.array([0.15, -0.1, 0.2])
    rotation, _ = cv2.Rodrigues(rvec)
    depth = 0.5
    tvec = np.array([(120 - 320) / 450 * depth, (100 - 240) / 450 * depth, depth]) - rotation @ center[0]
    k_true, d_true = views["camera_matrix"], views["dist_coeffs"]
    image = lens.render_plane(patch, origin, extent, k_true, d_true, rvec, tvec)
    raw = markers.detect(image)
    assert [d[0] for d in raw] == [10]
    distorted_truth, _ = cv2.projectPoints(center, rvec, tvec, k_true, d_true)
    assert np.linalg.norm(raw[0][1].mean(axis=0) - distorted_truth.reshape(2)) < 0.5
    found = markers.detect(lens.undistort(image, fitted))
    assert [d[0] for d in found] == [10]
    expected = cv2.undistortPoints(
        distorted_truth, fitted["camera_matrix"], fitted["dist_coeffs"], P=lens.new_camera_matrix(fitted)
    )
    assert np.linalg.norm(found[0][1].mean(axis=0) - expected.reshape(2)) < 0.5
    assert np.linalg.norm(found[0][1].mean(axis=0) - raw[0][1].mean(axis=0)) > 2.0  # the correction is real


def test_undistort_maps_match_undistort_and_keep_the_frame_size(views, fitted):
    image = _read(views["images"][0])
    map1, map2 = lens.undistort_maps(fitted)
    remapped = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    direct = lens.undistort(image, fitted)
    assert remapped.shape == direct.shape == (480, 640)
    assert np.mean(np.abs(remapped.astype(int) - direct.astype(int))) < 2.0
    color = lens.undistort(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), fitted)
    assert color.shape == (480, 640, 3)


def test_undistort_rejects_other_frame_sizes(fitted):
    with pytest.raises(ValueError, match="calibrated at 640x480"):
        lens.undistort(np.zeros((240, 320, 3), np.uint8), fitted)
    with pytest.raises(ValueError, match="frame"):
        lens.undistort(np.zeros((0, 0), np.uint8), fitted)


def test_check_image_reports_the_reprojection_rms(views, fitted):
    metrics = lens.check_image(_read(views["images"][2]), fitted, PATTERN, 0.025)
    assert metrics["corners"] == CORNERS and metrics["rms"] < 0.5 and metrics["accepted"] is True
    assert metrics["max_px"] >= metrics["rms"]
    with pytest.raises(ValueError, match="not found"):
        lens.check_image(np.full((480, 640), 200, np.uint8), fitted)


def test_parse_pattern_and_board_points():
    assert lens.parse_pattern("9x6") == (9, 6) and lens.parse_pattern("7X5") == (7, 5)
    for bad in ("9", "9x", "1x6", "2x6", "axb", "9x6x2"):
        with pytest.raises(ValueError):
            lens.parse_pattern(bad)
    pts = lens.board_points((3, 3), 0.1)
    assert pts.shape == (9, 3) and pts.dtype == np.float32
    assert np.allclose(pts[:4, :2], [[0, 0], [0.1, 0], [0.2, 0], [0, 0.1]])  # row by row
    assert np.allclose(pts[-1, :2], [0.2, 0.2]) and np.all(pts[:, 2] == 0)
    with pytest.raises(ValueError, match="positive"):
        lens.board_points((9, 6), 0.0)
    with pytest.raises(ValueError, match="3x3"):
        lens.board_points((2, 6), 0.025)


# -- CLI ------------------------------------------------------------------------------------------------


def test_cli_fit_check_and_the_synthetic_guard(views, tmp_path, monkeypatch, capsys):
    default = tmp_path / "config" / "lens.npz"
    monkeypatch.setattr(lens, "LENS_PATH", default)
    folder = views["images"][0].parent
    assert lens.main(["--images", str(folder)]) == 1
    assert not default.exists() and "synthetic images cannot calibrate" in capsys.readouterr().err
    out = tmp_path / "fit" / "lens.npz"
    assert lens.main(["--images", str(folder), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert (
        out.exists() and "used 18 of 18 images" in text and "rms" in text and "Synthetic views only" in text
    )
    assert lens.load_lens(out)["synthetic"] is True
    assert lens.main(["--check", str(views["images"][0]), "--lens", str(out)]) == 0
    assert "PASS" in capsys.readouterr().out
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((480, 640), 200, np.uint8))
    assert lens.main(["--check", str(blank), "--lens", str(out)]) == 1
    assert "lens check failed: the full checkerboard was not found" in capsys.readouterr().err
    assert lens.main(["--check", str(tmp_path / "missing.png"), "--lens", str(out)]) == 1
    assert "lens check failed" in capsys.readouterr().err
    assert lens.main(["--check", str(views["images"][0]), "--lens", str(tmp_path / "none.npz")]) == 1
    assert "no lens file" in capsys.readouterr().err


def test_cli_refuses_too_few_images_and_writes_nothing(views, tmp_path, capsys):
    folder = tmp_path / "few"
    folder.mkdir()
    for path in views["images"][:14]:
        (folder / path.name).write_bytes(path.read_bytes())
    out = tmp_path / "lens.npz"
    assert lens.main(["--images", str(folder), "--out", str(out)]) == 1
    assert not out.exists() and "need 15" in capsys.readouterr().err
    assert lens.main(["--images", str(tmp_path / "nowhere"), "--out", str(out)]) == 1
    assert not out.exists() and "not a directory" in capsys.readouterr().err


def test_cli_synthesize_never_overwrites_and_usage_errors_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")  # argparse wraps the --help text on a narrow terminal
    folder = tmp_path / "boards"
    assert lens.main(["--synthesize", str(folder)]) == 0
    assert len(list(folder.glob("synthetic-board-*.png"))) == lens.SYNTHETIC_VIEWS
    assert "true fx 450.00" in capsys.readouterr().out
    assert lens.main(["--synthesize", str(folder)]) == 1
    assert "nothing written" in capsys.readouterr().err
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "synthetic-board-05.png").write_bytes(b"keep")
    assert lens.main(["--synthesize", str(partial)]) == 1  # one taken name: refused before any write
    assert "synthetic-board-05.png" in capsys.readouterr().err
    assert [p.name for p in partial.iterdir()] == ["synthetic-board-05.png"]
    assert (partial / "synthetic-board-05.png").read_bytes() == b"keep"
    assert lens.main(["--check", str(tmp_path / "missing.png"), "--lens", str(tmp_path / "none.npz")]) == 1
    assert "no lens file" in capsys.readouterr().err
    for argv in (
        [],
        ["--images", str(folder), "--pattern", "9by6"],
        ["--images", str(folder), "--pattern", "2x6"],
        ["--images", str(folder), "--square-m", "0"],
        ["--images", str(folder), "--lens", str(tmp_path / "lens.npz")],
        ["--check", str(folder / "synthetic-board-01.png"), "--out", str(tmp_path / "lens.npz")],
        ["--synthesize", str(tmp_path / "more"), "--out", str(tmp_path / "lens.npz")],
        ["--images", str(folder), "--check", str(folder / "synthetic-board-01.png")],
    ):
        with pytest.raises(SystemExit) as exc:
            lens.main(argv)
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        lens.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert all(flag in text for flag in ("--images", "--check", "--synthesize", "--pattern", "--square-m"))
    assert "default config/lens.npz" in text and "default 0.025" in text
