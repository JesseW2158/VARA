"""Marker capture evidence and cleanup, using generated images, camera stubs and the FakeCamera only."""

from __future__ import annotations

import copy
import json

import cv2
import numpy as np
import pytest

from scripts import capture_table
from vision import calibrate, cameras, markers

SPOTS = [(0.12, -0.22), (0.12, 0.22), (0.28, 0.0), (0.44, -0.26), (0.44, 0.26)]
REAL_FAKE_CAMERA = cameras.FakeCamera  # kept before any test patches the module attribute


class StubCamera:
    is_fake = False
    index = 2

    def __init__(self, frame):
        self.frame = frame
        self.released = False

    def read(self):
        return self.frame

    def release(self):
        self.released = True


class SequenceCamera(StubCamera):
    """A stub that returns the next frame of a list on every read."""

    def __init__(self, frames):
        super().__init__(frames[0])
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        frame = self.frames[self.reads % len(self.frames)]
        self.reads += 1
        return frame


def frame_with(*placements):
    """A white 640x480 frame with markers pasted at (id, top row, left column, side px)."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    for marker_id, row, col, side in placements:
        patch = cv2.cvtColor(markers.generate(marker_id, side), cv2.COLOR_GRAY2BGR)
        frame[row : row + side, col : col + side] = patch
    return frame


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch, tmp_path):
    def refuse(_):
        pytest.fail("unmocked physical camera access")

    monkeypatch.setattr(capture_table.cameras, "get_camera", refuse)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_: False)
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    # Never read this laptop's runs/markers manifest or write the repo's measurements/.
    monkeypatch.setattr(capture_table, "DEFAULT_MANIFEST", tmp_path / "no-manifest" / "markers.json")
    monkeypatch.setattr(capture_table, "DEFAULT_CHECK_POINTS", tmp_path / "default" / "check-points.json")
    # Never undistort through this laptop's config/lens.npz: physical captures here run on raw frames.
    monkeypatch.setattr("vision.lens.LENS_PATH", tmp_path / "no-lens" / "lens.npz")


@pytest.fixture
def camera(monkeypatch):
    camera = StubCamera(frame_with((10, 100, 200, 60)))
    monkeypatch.setattr(capture_table.cameras, "get_camera", lambda _: camera)
    return camera


def spot_scene(x, y):
    """The fake scene with only the pill bottle's marker 10 lying flat on the table at (x, y)."""
    scene = copy.deepcopy(cameras.load_fake_scene())
    scene["objects"] = {
        "pill_bottle": {
            "marker": 10,
            "x": x,
            "y": y,
            "yaw": 0.0,
            "h": 0.0,
            "size": [0.035, 0.035],
            "seen": True,
        }
    }
    scene["obstacles"] = {}
    scene["hand"]["seen"] = False
    return scene


def use_fake_scene(monkeypatch, scene):
    """Make `--fake` captures render `scene` (read when each capture opens its FakeCamera)."""

    def fake_camera():
        camera = REAL_FAKE_CAMERA()
        camera.set_scene(scene)
        return camera

    monkeypatch.setattr(capture_table.cameras, "FakeCamera", fake_camera)


# -- the existing capture contract ---------------------------------------------------------------------


def test_physical_capture_preserves_image_and_reports_only_observed_pixels(camera, tmp_path):
    out = tmp_path / "spot-1"
    report = capture_table.capture(out)
    assert report == json.loads((out / "markers.json").read_text(encoding="utf-8"))
    assert report["mode"] == "physical" and report["camera_index"] == 2 and report["frames"] == 1
    (entry,) = report["markers"]
    assert entry["id"] == 10 and entry["pixel"] == pytest.approx([229.5, 129.5], abs=0.1)
    assert entry["edge_px"] == pytest.approx(60.0, abs=1.0) and entry["frames_seen"] == 1
    assert report["duplicate_ids"] == [] and "expected" not in report and "check_point" not in report
    (cx, cy), hc = calibrate.camera_pose()  # config/scene.yaml as it stands, placeholders or measured
    pose = report["camera_pose"]
    assert pose["Hc"] == hc and pose["c"] == [cx, cy] and pose["source"] == "config/scene.yaml"
    scene_text = calibrate.SCENE_PATH.read_text(encoding="utf-8")
    assert pose["placeholder"] is capture_table.camera_pose_placeholder(scene_text)
    assert np.array_equal(cv2.imread(str(out / report["original_image"])), camera.frame)
    assert not np.array_equal(cv2.imread(str(out / report["annotated_image"])), camera.frame)
    assert camera.released


@pytest.mark.parametrize("flag", ["FAKE", "FAKE_CAMERA"])
def test_implicit_fake_cannot_create_physical_evidence(monkeypatch, tmp_path, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="unset FAKE"):
        capture_table.capture(tmp_path / "refused")
    assert list(tmp_path.iterdir()) == []


def test_unexpected_fake_camera_is_closed_and_refused(camera, tmp_path):
    camera.is_fake = True
    with pytest.raises(ValueError, match="synthetic or unidentified"):
        capture_table.capture(tmp_path / "refused")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_explicit_fake_capture_is_labeled_even_without_fake_environment(tmp_path):
    report = capture_table.capture(tmp_path / "offline", fake=True)
    assert report["mode"] == "synthetic"
    assert report["original_image"].startswith("synthetic-")
    assert report["annotated_image"].startswith("synthetic-")
    assert report["markers"] and report["synthetic_hidden_calibration_ids"] == [6]


def test_existing_directory_is_never_overwritten_or_camera_opened(tmp_path):
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "markers.json").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        capture_table.capture(existing)
    assert (existing / "markers.json").read_text(encoding="utf-8") == "original"


def test_failed_read_releases_camera_and_does_not_create_evidence(camera, tmp_path, capsys):
    camera.frame = None
    assert capture_table.main(["--out", str(tmp_path / "failed")]) == 2
    assert camera.released and list(tmp_path.iterdir()) == []
    assert "returned no frame" in capsys.readouterr().err


def test_detection_failure_still_releases_camera(camera, monkeypatch, tmp_path):
    def fail(_):
        raise RuntimeError("detector failed")

    monkeypatch.setattr(capture_table.markers, "detect", fail)
    with pytest.raises(RuntimeError, match="detector failed"):
        capture_table.capture(tmp_path / "failed")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_encoding_failure_does_not_create_evidence(camera, monkeypatch, tmp_path):
    monkeypatch.setattr(capture_table.cv2, "imencode", lambda *_: (False, None))
    with pytest.raises(OSError, match="encode"):
        capture_table.capture(tmp_path / "failed")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_missing_markers_keeps_image_but_cli_returns_unsuccessful(camera, tmp_path, capsys):
    camera.frame[:] = 255
    out = tmp_path / "empty-view"
    assert capture_table.main(["--out", str(out)]) == 1
    assert json.loads((out / "markers.json").read_text(encoding="utf-8"))["markers"] == []
    assert camera.released
    assert "No marker centers available" in capsys.readouterr().out


def test_success_cli_prints_id_and_pixel(camera, tmp_path, capsys):
    assert capture_table.main(["--out", str(tmp_path / "spot-1")]) == 0
    output = capsys.readouterr().out
    assert "Saved physical frame" in output and "ID 10: pixel [229.5, 129.5] edge 59.8 px" in output


# -- frames and duplicate ids ---------------------------------------------------------------------------


def test_frames_take_the_median_corners_and_count_sightings(monkeypatch, tmp_path):
    steady = frame_with((10, 100, 200, 60))
    jumped = frame_with((10, 100, 230, 60))
    camera = SequenceCamera([steady, steady, jumped])
    monkeypatch.setattr(capture_table.cameras, "get_camera", lambda _: camera)
    report = capture_table.capture(tmp_path / "median", frames=3)
    (entry,) = report["markers"]
    assert entry["pixel"] == pytest.approx([229.5, 129.5], abs=0.1) and entry["frames_seen"] == 3
    assert report["frames"] == 3 and camera.reads == 3 and "median over 3 frames" in report["center_method"]
    assert np.array_equal(cv2.imread(str(tmp_path / "median" / report["original_image"])), jumped)
    with pytest.raises(ValueError, match="frames must be a positive integer"):
        capture_table.capture(tmp_path / "bad", frames=0)
    with pytest.raises(SystemExit) as exc:
        capture_table.main(["--out", str(tmp_path / "bad"), "--frames", "0"])
    assert exc.value.code == 2


def test_duplicate_ids_are_reported(camera, tmp_path, capsys):
    camera.frame = frame_with((10, 100, 200, 60), (10, 300, 400, 60))
    assert capture_table.main(["--out", str(tmp_path / "twice")]) == 0
    report = json.loads((tmp_path / "twice" / "markers.json").read_text(encoding="utf-8"))
    assert report["duplicate_ids"] == [10] and [m["id"] for m in report["markers"]] == [10]
    assert "seen more than once" in capsys.readouterr().out


def test_fake_capture_combines_frames_expect_all_and_printed_sides(monkeypatch, tmp_path, capsys):
    single = capture_table.capture(tmp_path / "single", fake=True)
    reads = []

    def noisy_fake_camera():
        camera = REAL_FAKE_CAMERA(noise=2.0, seed=3)  # seeded per-frame pixel noise: the median has work
        render = camera.read

        def read():
            reads.append(1)
            return render()

        camera.read = read
        return camera

    monkeypatch.setattr(capture_table.cameras, "FakeCamera", noisy_fake_camera)
    argv = ["--out", str(tmp_path / "frames"), "--fake", "--frames", "3", "--expect-all"]
    assert capture_table.main([*argv, "--printed-side-mm", "10=30.5", "30=40"]) == 1  # 6 under the box
    report = json.loads((tmp_path / "frames" / "markers.json").read_text(encoding="utf-8"))
    assert len(reads) == 3 and report["frames"] == 3 and report["mode"] == "synthetic"
    assert report["missing"] == [6] and "median over 3 frames" in report["center_method"]
    assert [e["id"] for e in report["markers"]] == [e["id"] for e in single["markers"]]
    assert all(entry["frames_seen"] == 3 for entry in report["markers"])
    for median, clean in zip(report["markers"], single["markers"], strict=True):
        assert median["pixel"] == pytest.approx(clean["pixel"], abs=0.5)
    assert report["printed_side_mm"] == [
        {"id": 10, "measured_mm": 30.5, "nominal_mm": 30.0},
        {"id": 30, "measured_mm": 40.0, "nominal_mm": 40.0},
    ]
    assert "ID 10: printed side 30.5 mm recorded (nominal 30 mm)" in capsys.readouterr().out


# -- --expect-all -----------------------------------------------------------------------------------------


def test_expected_ids_come_from_the_marker_table():
    assert capture_table.expected_ids() == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 30]
    assert capture_table.expected_ids(include_hand=True) == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 30, 40]


def test_expect_all_reports_the_hidden_calibration_marker_and_fails(tmp_path, capsys):
    report = capture_table.capture(tmp_path / "vis", fake=True, expect=capture_table.expected_ids())
    assert report["expected"] == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 30]
    assert report["missing"] == [6] and report["unexpected"] == []
    assert report["synthetic_hidden_calibration_ids"] == [6]
    assert report == json.loads((tmp_path / "vis" / "markers.json").read_text(encoding="utf-8"))
    assert capture_table.main(["--out", str(tmp_path / "cli"), "--fake", "--expect-all"]) == 1
    output = capsys.readouterr().out
    assert "Missing expected markers: [6]" in output
    assert "calibration marker(s) [6] lie under raised items" in output


def test_expect_all_passes_when_every_marker_is_visible(monkeypatch, tmp_path, capsys):
    scene = copy.deepcopy(cameras.load_fake_scene())
    scene["obstacles"]["cereal_box"].update({"x": 0.12, "y": 0.10})  # off calibration marker 6
    assert cameras.FakeCamera(scene).hidden_calibration_ids() == set()
    use_fake_scene(monkeypatch, scene)
    assert capture_table.main(["--out", str(tmp_path / "all"), "--fake", "--expect-all"]) == 0
    assert "All 12 expected markers detected." in capsys.readouterr().out
    report = json.loads((tmp_path / "all" / "markers.json").read_text(encoding="utf-8"))
    assert report["missing"] == [] and report["unexpected"] == []
    assert (
        capture_table.main(["--out", str(tmp_path / "hand"), "--fake", "--expect-all", "--expect-hand"]) == 1
    )
    assert "Missing expected markers: [40]" in capsys.readouterr().out
    # The wristband in view, clear of every marker (at the hand rest it would cover the juice box's 11).
    scene["hand"].update({"x": 0.36, "y": -0.20, "seen": True})
    assert REAL_FAKE_CAMERA(scene).hidden_calibration_ids() == set()
    report = capture_table.capture(tmp_path / "seen", fake=True, expect=capture_table.expected_ids())
    assert report["missing"] == [] and report["unexpected"] == [40]
    report = capture_table.capture(
        tmp_path / "seen-hand", fake=True, expect=capture_table.expected_ids(include_hand=True)
    )
    assert report["missing"] == [] and report["unexpected"] == []


def test_expect_hand_requires_expect_all(tmp_path):
    with pytest.raises(SystemExit) as exc:
        capture_table.main(["--out", str(tmp_path / "x"), "--fake", "--expect-hand"])
    assert exc.value.code == 2 and not (tmp_path / "x").exists()


# -- --printed-side-mm -------------------------------------------------------------------------------------


def write_manifest(path, entries):
    """A scripts/make_markers.py manifest: {"dpi", "markers": [{id, side_cm, ...}], "sheets"}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dpi": 300, "markers": list(entries), "sheets": []}), encoding="utf-8")


def test_printed_sides_parse_and_are_recorded_verbatim(camera, tmp_path, capsys):
    assert capture_table.parse_printed_sides(["10=30.5", "0=60"]) == {10: 30.5, 0: 60.0}
    for bad, message in (
        (["10"], "ID=MM"),
        (["ten=30"], "ID=MM"),
        (["10=0"], "positive"),
        (["10=nan"], "positive"),
        (["10=30", "10=31"], "twice"),
    ):
        with pytest.raises(ValueError, match=message):
            capture_table.parse_printed_sides(bad)
    report = capture_table.capture(tmp_path / "sides", printed_side_mm={10: 30.5, 0: 60.0})
    assert report["printed_side_mm"] == [
        {"id": 0, "measured_mm": 60.0, "nominal_mm": 60.0},
        {"id": 10, "measured_mm": 30.5, "nominal_mm": 30.0},
    ]
    assert report["nominal_side_source"].startswith("vision.markers sides by kind (no make_markers manifest")
    assert report == json.loads((tmp_path / "sides" / "markers.json").read_text(encoding="utf-8"))
    assert capture_table.main(["--out", str(tmp_path / "cli"), "--printed-side-mm", "10=30.5", "30=40"]) == 0
    output = capsys.readouterr().out
    assert "ID 10: printed side 30.5 mm recorded (nominal 30 mm)" in output
    assert "ID 30: printed side 40 mm recorded (nominal 40 mm)" in output
    assert "Nominal sides from vision.markers sides by kind" in output
    with pytest.raises(SystemExit) as exc:
        capture_table.main(["--out", str(tmp_path / "bad"), "--printed-side-mm", "10=abc"])
    assert exc.value.code == 2 and "ID=MM" in capsys.readouterr().err and not (tmp_path / "bad").exists()


def test_printed_sides_refuse_unknown_ids_and_unit_slips_before_the_camera_opens(camera, tmp_path, capsys):
    for measured, message in (
        ({99: 30.0}, "marker id 99 has no nominal side"),
        ({10: 3.0}, "did you type centimeters"),  # 3.0 cm typed for 30 mm
        ({0: 130.0}, "far from its nominal 60 mm"),
    ):
        with pytest.raises(ValueError, match=message):
            capture_table.capture(tmp_path / "refused", printed_side_mm=measured)
    assert not camera.released and not (tmp_path / "refused").exists()
    assert capture_table.main(["--out", str(tmp_path / "refused"), "--printed-side-mm", "10=3"]) == 2
    assert "Refused: printed side 3 mm of marker 10" in capsys.readouterr().err
    assert not camera.released and not (tmp_path / "refused").exists()
    with pytest.raises(ValueError, match="only supplies nominal sides"):
        capture_table.capture(tmp_path / "refused", manifest=tmp_path / "markers.json")
    with pytest.raises(SystemExit) as exc:
        capture_table.main(["--out", str(tmp_path / "x"), "--manifest", str(tmp_path / "markers.json")])
    assert exc.value.code == 2 and not (tmp_path / "x").exists()


def test_nominal_sides_come_from_the_make_markers_manifest(camera, monkeypatch, tmp_path, capsys):
    from scripts import make_markers  # what make_markers writes into its manifest, spare cards included

    manifest = tmp_path / "printed" / "markers.json"
    specs = [spec for spec in make_markers.marker_specs() if spec["id"] != 13]
    specs.append({"id": 13, "side_cm": 3.0, "name": "spare, I2 check", "kind": "object", "spare": True})
    write_manifest(manifest, specs)
    sides, source = capture_table.nominal_sides(manifest)
    assert sides[0] == 60.0 and sides[10] == 30.0 and sides[30] == 40.0 and sides[40] == 40.0
    assert sides[13] == 30.0 and source == manifest.resolve().as_posix()  # a spare card, not in objects.yaml
    report = capture_table.capture(tmp_path / "spare", printed_side_mm={13: 29.5}, manifest=manifest)
    assert report["printed_side_mm"] == [{"id": 13, "measured_mm": 29.5, "nominal_mm": 30.0}]
    assert report["nominal_side_source"] == source
    write_manifest(manifest, [{"id": 10, "side_cm": 3.5, "name": "pill_bottle", "kind": "object"}])
    with pytest.raises(ValueError, match=r"marker id 0 has no nominal side in .*printed/markers\.json"):
        capture_table.capture(tmp_path / "refused", printed_side_mm={0: 60.0}, manifest=manifest)
    monkeypatch.setattr(capture_table, "DEFAULT_MANIFEST", manifest)  # the default is read when present
    assert capture_table.main(["--out", str(tmp_path / "default"), "--printed-side-mm", "10=34.5"]) == 0
    output = capsys.readouterr().out
    assert "ID 10: printed side 34.5 mm recorded (nominal 35 mm)" in output
    assert f"Nominal sides from {source}." in output
    bad = tmp_path / "bad.json"
    for content, message in (
        ("junk", "not readable JSON"),
        (json.dumps({"markers": []}), "no markers list"),
        (json.dumps([{"id": 10, "side_cm": 3}]), "no markers list"),
        (json.dumps({"markers": [{"id": "10", "side_cm": 3}]}), "integer id"),
        (json.dumps({"markers": [{"id": 10, "side_cm": 0}]}), "positive side_cm"),
        (json.dumps({"markers": [{"id": 10, "side_cm": 3}, {"id": 10, "side_cm": 3}]}), "twice"),
    ):
        bad.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            capture_table.nominal_sides(bad)
    argv = ["--out", str(tmp_path / "missing"), "--printed-side-mm", "10=30", "--manifest"]
    assert capture_table.main([*argv, str(tmp_path / "missing.json")]) == 2
    assert "make_markers manifest" in capsys.readouterr().err and not (tmp_path / "missing").exists()


def test_camera_pose_placeholder_reads_the_header_and_the_camera_block():
    committed = (  # config/scene.yaml's header and camera block as committed before B4 measured them
        "# Measured once per table setup. Units: meters and radians. Placeholders until B4, B5 and A10 "
        "measure them.\n"
        "workspace: {x: [0.10, 0.40], y: [-0.25, 0.25], z: [0.02, 0.30]}\n"
        "camera:\n"
        "  Hc: 0.65            # lens height above the table (B4)\n"
        "  c: [0.20, -0.05]    # table point directly under the lens (B4)\n"
    )
    assert capture_table.camera_pose_placeholder(committed) is True
    measured = (
        "# Measured 2026-10-18 (B4).\n"
        "workspace: {x: [0.10, 0.40]}\n"
        "camera:\n"
        "  Hc: 0.712          # tape-measured lens height\n"
        "  c: [0.21, -0.04]\n"
        "calib_markers:\n"
        "  0: [0.08, 0.34]\n"
        "home_q: [0.0, -1.35]  # Placeholder until A10 agrees a pose\n"
    )
    assert capture_table.camera_pose_placeholder(measured) is False  # another block's note does not count
    assert capture_table.camera_pose_placeholder(measured.replace("tape-measured", "placeholder")) is True
    assert capture_table.camera_pose_placeholder("# Placeholders until B4.\n" + measured) is True
    assert capture_table.camera_pose_placeholder("camera: {Hc: 0.65, c: [0.2, -0.05]}  # PLACEHOLDER\n")


# -- --check-point --------------------------------------------------------------------------------------


def test_check_point_appends_the_pixel_and_its_provenance(camera, tmp_path, capsys):
    path = tmp_path / "measurements" / "check-points.json"
    out = tmp_path / "spot-1"
    point = capture_table.CheckPoint((0.21, -0.08), 10, path)
    report = capture_table.capture(out, check_point=point)
    block = report["check_point"]
    assert (
        block["appended"] is True and block["index"] == 0 and block["count"] == 1 and block["reason"] is None
    )
    assert block["spot_id"] == 10 and block["table"] == [0.21, -0.08] and block["file"] == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"pixel": pytest.approx([229.5, 129.5], abs=0.1), "table": [0.21, -0.08]}
    ]
    provenance = json.loads(capture_table.provenance_path(path).read_text(encoding="utf-8"))
    assert len(provenance) == 1 and provenance[0]["index"] == 0 and provenance[0]["spot_id"] == 10
    assert provenance[0]["table"] == [0.21, -0.08] and provenance[0]["mode"] == "physical"
    assert provenance[0]["image"] == (out / "physical-original.png").resolve().as_posix()
    assert provenance[0]["t"] == report["captured_at"]
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "check-points.json",
        "check-points.provenance.json",
    ]
    camera.frame = frame_with((10, 200, 300, 60))
    argv = ["--out", str(tmp_path / "spot-2"), "--check-point", "0.30", "-0.10", "--append", str(path)]
    assert capture_table.main(argv) == 0
    assert "Check point 2 appended" in capsys.readouterr().out
    assert [p["table"] for p in capture_table.load_check_point_list(path)] == [[0.21, -0.08], [0.3, -0.1]]


def test_check_point_defaults_to_the_measurements_list(camera, tmp_path, capsys):
    default = capture_table.DEFAULT_CHECK_POINTS  # under tmp_path (the autouse fixture)
    assert not default.parent.exists()
    assert capture_table.main(["--out", str(tmp_path / "spot-1"), "--check-point", "0.21", "-0.08"]) == 0
    assert f"appended to {default}" in capsys.readouterr().out
    (point,) = capture_table.load_check_point_list(default)
    assert point["table"] == [0.21, -0.08] and point["pixel"] == pytest.approx([229.5, 129.5], abs=0.1)
    report = json.loads((tmp_path / "spot-1" / "markers.json").read_text(encoding="utf-8"))
    assert report["check_point"]["spot_id"] == 10 and report["check_point"]["file"] == str(default)


def test_check_point_refusals_before_the_camera_opens(camera, tmp_path, capsys):
    path = tmp_path / "check-points.json"
    path.write_text(json.dumps([{"pixel": [229.5, 129.5], "table": [0.21, -0.08]}]), encoding="utf-8")
    cases = [
        (capture_table.CheckPoint((0.21, -0.08), 10, path), False, "already entry 0"),
        (capture_table.CheckPoint((0.3, 0.1), 10, path), True, "synthetic-"),
        (
            capture_table.CheckPoint((0.3, 0.1), 10, tmp_path / "synthetic-list.json"),
            False,
            "labeled synthetic",
        ),
        (capture_table.CheckPoint((0.3, 0.1), 3, path), False, "fitting marker"),
        (capture_table.CheckPoint((0.3, 0.1), 50, path), False, "0-49"),
        (capture_table.CheckPoint((21.0, -8.0), 10, path), False, "centimeters"),
        (capture_table.CheckPoint((float("nan"), 0.1), 10, path), False, "finite"),
        (capture_table.CheckPoint((0.3, 0.1), 10, tmp_path / "list.txt"), False, ".json"),
    ]
    for point, fake, message in cases:
        with pytest.raises(ValueError, match=message):
            capture_table.capture(tmp_path / "refused", fake=fake, check_point=point)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("junk", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        capture_table.capture(
            tmp_path / "refused", check_point=capture_table.CheckPoint((0.3, 0.1), 10, corrupt)
        )
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([{"pixel": [1, 2], "table": [0.1, 0.2], "spot_id": 10}]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly pixel"):
        capture_table.capture(
            tmp_path / "refused", check_point=capture_table.CheckPoint((0.3, 0.1), 10, extra)
        )
    argv = ["--out", str(tmp_path / "refused"), "--check-point", "0.21", "-0.08", "--append", str(path)]
    assert capture_table.main(argv) == 2
    assert "Refused: table point [0.21, -0.08] is already entry 0" in capsys.readouterr().err
    assert not camera.released and not (tmp_path / "refused").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"pixel": [229.5, 129.5], "table": [0.21, -0.08]}]


def test_a_stale_or_malformed_provenance_sidecar_is_refused_before_the_camera_opens(camera, tmp_path):
    path = tmp_path / "check-points.json"
    sidecar = capture_table.provenance_path(path)
    assert sidecar.name == "check-points.provenance.json"
    sidecar.write_text("[]", encoding="utf-8")  # left behind by a deleted list: two sessions would mix
    point = capture_table.CheckPoint((0.3, 0.1), 10, path)
    with pytest.raises(ValueError, match="exists without check-points.json"):
        capture_table.capture(tmp_path / "refused", check_point=point)
    path.write_text("[]", encoding="utf-8")
    for content in ("junk", "{}", "[1, 2]"):
        sidecar.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="provenance"):
            capture_table.capture(tmp_path / "refused", check_point=point)
    assert not camera.released and not (tmp_path / "refused").exists()


def test_check_point_refusals_after_the_capture_keep_the_images(camera, tmp_path, capsys):
    path = tmp_path / "check-points.json"
    argv = ["--check-point", "0.30", "0.10", "--append", str(path)]
    assert capture_table.main(["--out", str(tmp_path / "absent"), *argv, "--spot-id", "11"]) == 1
    assert "NOT appended: marker 11 not detected" in capsys.readouterr().err
    report = json.loads((tmp_path / "absent" / "markers.json").read_text(encoding="utf-8"))
    assert report["check_point"]["appended"] is False and "not detected" in report["check_point"]["reason"]
    assert (tmp_path / "absent" / "physical-original.png").exists() and not path.exists()
    camera.frame = frame_with((10, 100, 200, 60), (10, 300, 400, 60))
    assert capture_table.main(["--out", str(tmp_path / "twice"), *argv]) == 1
    assert "detected 2 times" in capsys.readouterr().err and not path.exists()
    camera.frame = frame_with((10, 100, 200, 60))
    assert capture_table.main(["--out", str(tmp_path / "first"), *argv]) == 0
    capsys.readouterr()
    assert (
        capture_table.main(
            ["--out", str(tmp_path / "same"), "--check-point", "0.31", "0.11", "--append", str(path)]
        )
        == 1
    )
    assert "did not move" in capsys.readouterr().err
    assert len(capture_table.load_check_point_list(path)) == 1


def test_five_fake_spots_pass_the_offline_check(monkeypatch, tmp_path, capsys):
    path = tmp_path / "synthetic-check-points.json"
    for index, (x, y) in enumerate(SPOTS, start=1):
        use_fake_scene(monkeypatch, spot_scene(x, y))
        argv = [
            "--out",
            str(tmp_path / f"spot-{index}"),
            "--fake",
            "--check-point",
            str(x),
            str(y),
            "--append",
            str(path),
        ]
        assert capture_table.main(argv) == 0
        output = capsys.readouterr().out
        assert f"{index} of 5 recorded" in output
        assert ("check points valid" in output) == (index == 5)
    points = capture_table.load_check_point_list(path)
    assert [p["table"] for p in points] == [list(spot) for spot in SPOTS]
    provenance = json.loads(capture_table.provenance_path(path).read_text(encoding="utf-8"))
    assert [p["mode"] for p in provenance] == ["synthetic"] * 5 and [p["index"] for p in provenance] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert calibrate.main(["--check", "--check-points", str(path)]) == 0
    assert "PASS" in capsys.readouterr().out
    # The synthetic camera and the synthetic homography agree to sub-pixel detection noise (1 px is
    # about 1.4 mm on the table), far inside the 1.5 cm limit.
    assert calibrate.check_points_metrics(calibrate.load_homography(), points)["max_cm"] < 0.25


def test_fake_camera_check_point_refusals_leave_the_list_unchanged(monkeypatch, tmp_path, capsys):
    path = tmp_path / "synthetic-check-points.json"
    use_fake_scene(monkeypatch, spot_scene(*SPOTS[0]))

    def run(name, x, y, *extra, append=path):
        argv = [
            "--out",
            str(tmp_path / name),
            "--fake",
            "--check-point",
            str(x),
            str(y),
            "--append",
            str(append),
        ]
        status = capture_table.main([*argv, *extra])
        captured = capsys.readouterr()
        return status, captured.out + captured.err

    assert run("first", *SPOTS[0])[0] == 0
    # Refused before the camera opens (exit 2, no capture directory): the same table point again, and a
    # synthetic capture aimed at a list without the synthetic- label.
    status, text = run("again", *SPOTS[0])
    assert status == 2 and "already entry 0" in text and not (tmp_path / "again").exists()
    status, text = run("unlabeled", 0.30, 0.10, append=tmp_path / "check-points.json")
    assert status == 2 and "synthetic-" in text and not (tmp_path / "unlabeled").exists()
    # Refused after the capture (exit 1, the synthetic evidence stays): the marker did not move, the spot
    # marker is not in view, and two copies of the spot marker are in view.
    status, text = run("unmoved", 0.13, -0.21)
    assert status == 1 and "did not move" in text
    status, text = run("absent", 0.30, 0.10, "--spot-id", "11")
    assert status == 1 and "marker 11 not detected" in text
    twice = spot_scene(*SPOTS[1])
    twice["objects"]["spare"] = {**twice["objects"]["pill_bottle"], "x": SPOTS[2][0], "y": SPOTS[2][1]}
    use_fake_scene(monkeypatch, twice)
    status, text = run("twice", 0.30, 0.10)
    assert status == 1 and "detected 2 times" in text
    for name in ("unmoved", "absent", "twice"):
        report = json.loads((tmp_path / name / "markers.json").read_text(encoding="utf-8"))
        assert report["mode"] == "synthetic" and report["check_point"]["appended"] is False
        assert report["check_point"]["reason"] and (tmp_path / name / "synthetic-original.png").exists()
    (only,) = capture_table.load_check_point_list(path)
    assert only["table"] == list(SPOTS[0])
    assert len(json.loads(capture_table.provenance_path(path).read_text(encoding="utf-8"))) == 1
    assert not (tmp_path / "check-points.json").exists()


def test_a_check_point_without_a_path_goes_to_the_default_list_of_the_moment(camera, tmp_path):
    assert capture_table.CheckPoint((0.21, -0.08)).path is None
    report = capture_table.capture(tmp_path / "spot-1", check_point=capture_table.CheckPoint((0.21, -0.08)))
    default = capture_table.DEFAULT_CHECK_POINTS  # patched to tmp_path by the autouse fixture
    assert report["check_point"]["appended"] is True and report["check_point"]["file"] == str(default)
    assert [p["table"] for p in capture_table.load_check_point_list(default)] == [[0.21, -0.08]]


def test_cli_help_lists_every_flag(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")  # argparse wraps at hyphens on a narrow terminal
    with pytest.raises(SystemExit) as exc:
        capture_table.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--out",
        "--fake",
        "--frames",
        "--expect-all",
        "--expect-hand",
        "--printed-side-mm",
        "--manifest",
        "--check-point",
        "--spot-id",
        "--append",
    ):
        assert flag in text
    assert "measurements/check-points.json" in text


def test_check_point_usage_errors(tmp_path):
    for argv in (
        ["--out", str(tmp_path / "x"), "--spot-id", "10"],
        ["--out", str(tmp_path / "x"), "--append", str(tmp_path / "l.json")],
        ["--out", str(tmp_path / "x"), "--check-point", "0.1"],
        ["--out", str(tmp_path / "x"), "--check-point", "a", "b"],
    ):
        with pytest.raises(SystemExit) as exc:
            capture_table.main(argv)
        assert exc.value.code == 2
    assert not (tmp_path / "x").exists()
