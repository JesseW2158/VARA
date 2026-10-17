# Calibrate the actual table

The committed `config/homography.npy` describes the synthetic camera. It does not calibrate the laptop
webcam or the physical table. B5 remains incomplete until measurements from the mounted camera pass
the independent checks below. This procedure reads cameras and writes a matrix; it never moves the arm.

1. Mount the overhead camera rigidly and identify its real device index. The laptop's built-in webcam
   is not an overhead camera. Set `OVERHEAD_CAM` in `.env`, and record the camera's measured lens height
   `Hc` and table position `c` in `config/scene.yaml`. Use the robot's coordinate frame: meters, origin under the shoulder
   pan axis, +x forward, +y left. Complete the physical marker/mount steps in B3 and B4 first.
2. Place markers 0–7 around the workspace, keeping them visible and clear of objects. Tape-measure their
   actual centers into `config/scene.yaml` under `calib_markers`. The existing coordinates are placeholders.
3. Select at least five additional spots spread across the usable table, including its edges and center.
   Use spots other than the eight fitting markers. For each, record the center pixel `[u, v]` from an
   overhead image and independently tape-measure its table coordinates `[x, y]` in meters at z=0.
   Keep the original image and measurement notes with this setup's evidence.

Use a loose printed object marker such as ID 10 to identify each check spot. Lay it flat directly on the
table, without its object underneath, and keep other copies of that ID out of view. Move this same marker
to five separately tape-measured positions; do not reuse the fitting markers 0-7. Keep the camera and the
fitting markers fixed throughout. Close other programs using the overhead camera, then capture each spot:

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/table-calibration/spot-1
```

Repeat with new directories `spot-2` through `spot-5`. The command uses `OVERHEAD_CAM` from `.env`, never
moves the arm, and refuses an existing output directory. Each directory contains the untouched
`physical-original.png`, an annotated image with marker IDs and center crosses, and `markers.json`.
Copy ID 10's observed `pixel` pair from that report and combine it with your separately measured `table`
pair below. The center is the mean of the four detected corners, in pixels from the image's top-left
(`u` right, `v` down). These are image measurements only; the tool does not infer table coordinates or
save a homography. With no detections it keeps the image for diagnosis and exits unsuccessfully.
`--fake` is available for offline checks, but its files are explicitly labeled synthetic and cannot serve
as physical calibration evidence.

Save those measurements as a JSON list in `measurements/check-points.json`. Each entry has this shape
(the numbers below illustrate the format only; replace them with your measurements):

```json
{"pixel": [123.4, 234.5], "table": [0.21, -0.08]}
```

The file needs at least five entries, with distinct pixel and table positions spanning two dimensions.
Remove `FAKE` and `FAKE_CAMERA` from the calibration process environment; camera modes reject a synthetic
camera so a fake setting cannot silently replace the physical calibration.

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
response on success and an error if fewer than four markers are visible or either limit fails. It does
not replace the matrix or certify the independent five-point measurements. The dashboard's fake mode
still returns its labeled simulation fixture.

Repeat measurements after the camera, mount, table reference or marker layout changes. Recorded pixel
checks cannot detect a camera move made after their source image. If edge errors remain high, complete
the lens calibration/undistortion described in B5 before attempting robot execution. Restart the
orchestrator after saving a new matrix so the tracker loads it.
