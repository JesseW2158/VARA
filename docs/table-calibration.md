# Calibrate the actual table

The committed `config/homography.npy` describes the synthetic camera. It does not calibrate the laptop
webcam or the physical table. B5 remains incomplete until measurements from the mounted camera pass
the independent checks below. This procedure reads cameras and writes a matrix; it never moves the arm.

1. Mount the overhead camera rigidly and identify its real device index. The laptop's built-in webcam
   is not an overhead camera. Set `OVERHEAD_CAM` in `.env`, and record the camera's measured lens height
   `Hc` and table position `c` in `config/scene.yaml`. Use the robot's coordinate frame: meters, origin under the shoulder
   pan axis, +x forward, +y left. Complete the physical marker/mount steps in B3 and B4 first, and confirm
   with the visibility capture below that the mounted camera sees every marker.
2. Place markers 0–7 around the workspace, keeping them visible and clear of objects. Tape-measure their
   actual centers into `config/scene.yaml` under `calib_markers`. The existing coordinates are placeholders.
3. Select at least five additional spots spread across the usable table, including its edges and center.
   Use spots other than the eight fitting markers. For each, record the center pixel `[u, v]` from an
   overhead image and independently tape-measure its table coordinates `[x, y]` in meters at z=0.
   `scripts/capture_table.py --check-point` reads that pixel from the capture and records the pair for
   you (see Record the check spots). Keep the original image and measurement notes with this setup's
   evidence.

## Confirm every marker is visible (B3, B4)

B3 is done when every `config/objects.yaml` marker plus the eight calibration markers is printed, measured
and detected from the camera's height; B4 when `Hc` and `c` are in `config/scene.yaml` and a frame with
the arm at home shows all markers. One capture records both. Have the arm at `home_q` first (this
command never moves it), every object on its taped spot marker side up, the cereal box on its tape,
markers 0-7 uncovered and the hand rest empty. Close other programs using the overhead camera, then:

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-1 --expect-all --frames 10
```

`--expect-all` compares the detected ids with every id the overhead camera is expected to see: the
calibration markers 0-7 and every `config/objects.yaml` id (the objects 10, 11 and 12 and the cereal box's
obstacle marker 30), not the wristband. `markers.json` gains `expected`, `missing` and `unexpected`; the
command prints the missing ids and exits 1 when any is missing. Add `--expect-hand` to require the
wristband's 40 as well (wear it in view, clear of every other marker). `--frames 10` takes each marker's
median corners over ten reads, like the verification snapshots; every entry also records its mean edge
length `edge_px` and `frames_seen`, and ids detected twice in one frame are listed under `duplicate_ids`.

`markers.json` also records `camera_pose`: `Hc` and `c` copied from `config/scene.yaml`, and
`placeholder`, true while that file's leading comment or its `camera:` block still calls the values
placeholders. A capture with `placeholder: true` is not B4 evidence: measure `Hc` and `c`, update the
values and the comment, and capture again in a new directory.

Record each marker's ruler-measured printed side (B3's "measured") with the same capture:

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-2 --expect-all --frames 10 --printed-side-mm 0=60.0 10=30.0 30=40.0
```

The numbers illustrate the format only; give every marker you measured, in millimeters as the ruler
reads. Each value is stored verbatim under `printed_side_mm` as `{"id", "measured_mm", "nominal_mm"}`.
The nominal side comes from the `scripts/make_markers.py` manifest (`runs/markers/markers.json`, its
default output; `--manifest PATH` names another) or, without a manifest, from the sides make_markers
prints by kind; `nominal_side_source` says which. An id the manifest lacks, or a value under half or over
twice its nominal side (centimeters typed as millimeters), is refused before the camera opens.

## Record the check spots

Use a loose printed object marker such as ID 10 to identify each check spot. Lay it flat directly on the
table, without its object underneath, and keep other copies of that ID out of view. Move this same marker
to five separately tape-measured positions; do not reuse the fitting markers 0-7. Keep the camera and the
fitting markers fixed throughout. Close other programs using the overhead camera, then tape-measure the
marker's center and capture the spot with that table point in meters (`X Y`; the numbers below
illustrate the format only):

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/table-calibration/spot-1 --check-point 0.21 -0.08
```

Repeat with new directories `spot-2` through `spot-5`, each with its own measured `--check-point X Y`.
The command uses `OVERHEAD_CAM` from `.env`, never moves the arm, and refuses an existing output
directory. Each directory contains the untouched `physical-original.png`, an annotated image with marker
IDs and center crosses, and `markers.json`. The center is the mean of the four detected corners, in
pixels from the image's top-left (`u` right, `v` down). These are image measurements only; the tool does
not infer table coordinates or save a homography. With no detections it keeps the image for diagnosis
and exits unsuccessfully. `--fake` is available for offline checks, but its files are explicitly labeled
synthetic and cannot serve as physical calibration evidence.

`--check-point X Y` takes marker 10's observed `pixel` from this capture and appends it with your
measured `table` pair to `measurements/check-points.json` (created with its folder on first use), so no
pixel is copied by hand. `--spot-id ID` uses another loose marker (never 0-7); `--append PATH` writes
another list (`POST /calibrate` reads the file named by `VARA_CHECK_POINTS` when that is set, so append
to that file then). The list keeps exactly the `pixel` and `table` pair per entry that
`vision.calibrate --check-points` and `POST /calibrate` accept; the full record of each entry (spot id,
source image, capture time, mode, `lens`) goes to the sidecar `measurements/check-points.provenance.json`, and
the spot's `markers.json` records the outcome under `check_point`. The command prints `N of 5 recorded`;
from the fifth entry on it validates the whole list and prints `check points valid` (or what is still
wrong, such as five spots on one line).

Refused before the camera opens (exit 2, nothing written): a table point already in the list, a
coordinate beyond 2 m (centimeters typed as meters), a spot id of 0-7 or outside 0-49, a list that is not
a JSON list of `pixel`/`table` pairs, a provenance sidecar that is malformed or left behind without its
list or whose records name another lens than this capture's, a corrupt `config/lens.npz`, and with
`--fake` any list not named `synthetic-*.json` (without `--fake`, a list named so).
Refused after the capture (exit 1; the images and `markers.json` stay, with the reason under
`check_point.reason`, and the list is unchanged): the spot marker not detected, detected more than once,
or within 0.5 px of a listed pixel (the marker did not move). To start a new list, move both
`check-points.json` and `check-points.provenance.json` out of `measurements/`.

The list is a JSON list in `measurements/check-points.json`. Each entry has this shape (the numbers
below illustrate the format only; replace them with your measurements):

```json
{"pixel": [123.4, 234.5], "table": [0.21, -0.08]}
```

The file needs at least five entries, with distinct pixel and table positions spanning two dimensions.
A list written by hand in this format is checked the same way. Remove `FAKE` and `FAKE_CAMERA` from the
calibration process environment; camera modes reject a synthetic camera so a fake setting cannot
silently replace the physical calibration.

## Fit, check and save the homography

Preview a new fit and both checks without replacing the existing matrix:

```powershell
.\.venv\Scripts\python.exe -m vision.calibrate --live --no-save --check-points measurements/check-points.json
```

After inspecting the reported residuals, save the accepted fit:

```powershell
.\.venv\Scripts\python.exe -m vision.calibrate --live --check-points measurements/check-points.json
```

Every check requires **mean error at most 1.0 cm and maximum error at most 1.5 cm**. Fewer than four known
fitting markers, invalid measurements, rejected residuals or a failed write leave the previous file
unchanged. A fit without `--check-points` may be saved after its marker residuals pass, but its independent
table accuracy is explicitly reported as unverified. A low error on the same markers used to fit the
matrix is not independent accuracy evidence.

The two checks answer different questions:

```powershell
# Check current fitting markers against the saved matrix using the overhead camera.
.\.venv\Scripts\python.exe -m vision.calibrate --check

# Check the saved matrix against recorded independent points; does not open a camera or save anything.
.\.venv\Scripts\python.exe -m vision.calibrate --check --check-points measurements/check-points.json
```

`POST /calibrate` performs the current marker consistency check. It returns the existing `error_cm`
response on success (with `check_points` and `lens`, the lens line's value; see Lens calibration below)
and an error if fewer than four markers are visible or either limit fails. It does
not replace the matrix or certify the independent five-point measurements. The dashboard's fake mode
still returns its labeled simulation fixture.

Repeat measurements after the camera, mount, table reference or marker layout changes. Recorded pixel
checks cannot detect a camera move made after their source image. If edge errors remain high, complete
the lens calibration and undistortion below (B5) before attempting robot execution. Restart the
orchestrator after saving a new matrix so the tracker loads it.

## Lens calibration (B5 fallback)

Eight spread fitting markers average out most lens distortion, so calibrate the lens only when the check
spots near the table edges still read over 1.5 cm off after a careful refit. It reads camera images
only and never moves the arm.

1. Print a flat checkerboard of 10 x 7 squares (9 x 6 inner corners, the default `--pattern 9x6`) and
   measure one square with a ruler; the default `--square-m 0.025` is 25 mm. The square size scales only
   the board poses, not the lens model, but give the measured value.
2. Photograph the board with the overhead camera at least 15 times: flat, entirely in view, moved and
   tilted so its corners together cover the whole frame, the edges and corners above all (distortion
   grows there). One capture directory per photo keeps the tracker's camera mode and an untouched PNG:

   ```powershell
   .\.venv\Scripts\python.exe scripts/capture_table.py --out runs/lens/board-01
   ```

   Repeat with `board-02` and onward. The fit reads each directory's `physical-original.png` and ignores
   the annotated copies; when no ArUco marker is in view the command still saves the frame and exits 1,
   which is fine here. Take one more photo outside that folder for the check, for example
   `--out runs/lens-check/board`.
3. Fit and save:

   ```powershell
   .\.venv\Scripts\python.exe -m vision.lens --images runs/lens
   ```

   It finds the inner corners (`cv2.findChessboardCorners`, refined to sub-pixel), fits
   `cv2.calibrateCamera` and writes `config/lens.npz` atomically: image size, camera matrix, distortion
   coefficients, reprojection RMS, per-image RMS and a hash of the photos used (`--out PATH` writes a
   trial elsewhere). Photos without the full board and byte-identical repeats are skipped and listed.
   Fewer than 15 usable photos or an RMS above 1 px is refused and writes nothing; retake sharper,
   flatter photos.
4. Check the lens on the photo the fit did not use:

   ```powershell
   .\.venv\Scripts\python.exe -m vision.lens --check runs/lens-check/board/physical-original.png
   ```

   It prints the reprojection RMS and exits 0 at 1 px or less, 1 otherwise (`--lens PATH` checks a trial
   file).

Once `config/lens.npz` exists, these tools undistort every overhead frame before detecting markers
(`vision.calibrate.load_active_lens` loads the file once, before the camera opens, and
`undistort_frame` remaps each frame with the optimal new camera matrix, keeping the whole field of view):
`vision.calibrate --live` and `--check`, `scripts/capture_table.py`, the scene tracker (the orchestrator,
`vision.verify` and `python -m vision.scene`; the dashboard's overhead stream then shows the undistorted
frame) and the dashboard's `POST /calibrate` check, which uses the tracker's lens. `capture_table.py`
still saves `physical-original.png` raw, so board photos for a new fit stay valid; its pixels, annotated
image and check points are undistorted. Without the file every tool runs on raw frames, as before. A
corrupt or incomplete file stops each tool at startup (exit 2, or a 503 from `/calibrate`); it never
falls back to raw frames silently. A frame whose size differs from the lens's is refused, so calibrate
the lens in the 640x480 mode the tracker uses; to photograph a board in another mode, move
`config/lens.npz` aside first.

Raw and undistorted pixels do not mix. After creating or replacing `config/lens.npz`:

1. Refit and save the homography with `python -m vision.calibrate --live` (above).
2. Move `measurements/check-points.json` and its `.provenance.json` sidecar away and capture the five
   check spots again with `capture_table.py --check-point`. It refuses to append to a list whose
   provenance records name another lens (records written before the lens wiring count as raw).
3. Check with `--check --check-points`, then restart the orchestrator so its tracker loads the new lens.

Reading the `lens:` line: calibrate (`--live`, `--check`, `--check --check-points`), `capture_table.py`
and `vision.verify` each print one. `lens: <64 hex digits>` is the sha256 of the `config/lens.npz` every
frame was undistorted with; `lens: null` means raw frames (no lens file, or a synthetic camera, which is
an ideal pinhole and never undistorted). The same value is `lens` in capture_table's `markers.json` and in
each check-points provenance record, `lens` in the `POST /calibrate` reply, and
`tracker_diagnostics.lens` in `vision.verify`'s `session.json` and `result.json`. Every piece of evidence
in one calibration must show the same value; compare it with `Get-FileHash config\lens.npz` (SHA256)
when in doubt.

Offline, `python -m vision.lens --synthesize DIR` writes 18 synthetic board photos of a known camera (the
synthetic pinhole with k1 = -0.10) for exercising the fit. They are named `synthetic-*`, a fit over them
is labeled synthetic, and writing that fit to the default `config/lens.npz` is refused (pass `--out`
elsewhere). The FakeCamera is an ideal pinhole and never needs undistortion.
